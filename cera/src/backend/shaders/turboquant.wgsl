// TurboQuant KV compression kernels (arXiv:2504.19874) — GPU port of the encode
// half of `cera/src/turboquant.rs`.
//
// Three entry points:
//   tq_encode_keys   — 2-bit PolarQuant + 1-bit QJL residual + f16 norms
//   tq_encode_values — 2-bit PolarQuant + f16 norm (no residual; the value read
//                      path is a weighted sum, not an inner product, so the JL
//                      sign-bit estimator doesn't apply)
//   tq_rotate_q      — pre-rotate query heads for the compressed-KV attention
//                      kernel (q_rot, q_jl, and the per-head sum of q_jl)
//
// Every entry point does a Randomized Hadamard Transform in workgroup memory:
// apply the layer's random ±1 sign flips, run an in-place Walsh-Hadamard
// butterfly, scale by 1/sqrt(head_dim). The CPU's `rht_forward` fuses the sign
// flip into the first butterfly stage; doing it as a separate pass first is
// mathematically identical and parallelizes cleanly.
//
// ── Cache layout ───────────────────────────────────────────────────────────
// Both cache buffers are `array<u32>` split into contiguous REGIONS rather than
// per-timestep interleaved records, so (a) each stream stays contiguous across
// timesteps — adjacent threads in the attention tile read adjacent words — and
// (b) a host-side snapshot readback of one head's stream is a straight copy.
//
//   keys:   [polar | jl | norms]
//     polar word w of (kv_head h, timestep t):
//        (h * max_seq_len + t) * (head_dim/16) + w
//     jl word w:      jl_off   + (h * max_seq_len + t) * (head_dim/32) + w
//     norms:          norm_off + (h * max_seq_len + t)
//        one u32 = pack2x16float(vec2(norm, residual_norm))
//   values: [polar | norms]
//     norms: one u32 = pack2x16float(vec2(norm, 0.0)); the high half is padding.
//
// Region offsets are derived from (n_kv_heads, max_seq_len, head_dim) — nothing
// extra is passed in.
//
// 2 bits per element, 16 elements per u32, element j at bit 2j (LSB-first); JL
// signs are 1 bit per element, 32 per u32, element j at bit j. On a little-endian
// host that makes a `bytemuck::cast_slice::<u32, u8>` of a readback
// byte-identical to the CPU's `pack_2bit` / `pack_1bit` output, which is what
// lets `encode_compressed_keys` consume it unchanged.
//
// Norms are rounded through `pack2x16float` so the GPU cache holds exactly the
// f16-rounded value the CPU stores in `CompressedKeyCache::norms` — the encode
// itself still uses the full-precision norm, matching the CPU.
//
// ── Constraints (asserted host-side) ───────────────────────────────────────
//   - head_dim is a power of two (Walsh-Hadamard), <= 128 (bounds the workgroup
//     arrays), and a multiple of 32 (whole JL sign words).
//
// ── Why this file repeats what turboquant.metal factors out ────────────────
// The MSL port hoists the tree reduction and the Walsh-Hadamard butterfly into
// `tq_reduce_sum` / `tq_wht`. This file deliberately inlines both: naga's SPIR-V
// path (lavapipe/Vulkan) miscompiles `workgroupBarrier()` reached through a
// function call inside a loop, and the butterfly's barrier is exactly that. The
// same constraint is why `flash_attention.wgsl` and the gemv kernels keep every
// barrier at entry-point scope. Metal has no such bug, so the MSL side is free to
// factor — and does, which is also why the two files' line counts differ.
//
// ── Uniform control flow ───────────────────────────────────────────────────
// Reads from `var<workgroup>` are non-uniform under WGSL's uniformity analysis,
// so no `workgroupBarrier()` may sit inside a branch predicated on one. The
// zero-vector short-circuit the CPU writes as an early `continue` is therefore
// expressed here as `select(...)` at the three write sites instead of a `return`:
// the rotation runs unconditionally on a safe divisor, and a zero-norm vector
// simply has its packed words forced to 0. Same reasoning for the QJL branch.
//
// Bind group 0 (all three entry points):
//   @binding(0) src:       array<f32>  (n_tokens x src_stride source rows)
//   @binding(1) out_words: array<u32>  (read_write; the f32 outputs of
//                                       tq_rotate_q are `bitcast<u32>`, so one
//                                       binding type serves all three kernels)
//   @binding(2) signs:     array<f32>  (all layers: [polar | jl] per layer)
//   @binding(3) params:    TqParams
// Dispatch: n_tokens * n_heads workgroups, 2-D and recovered via `get_wid`.

#include "common_decls.tmpl"

// Mirrors the Rust `TqParams` (single source of truth for this layout).
struct TqParams {
    // Rows in this batch: prefill tokens, or 1 for decode.
    n_tokens: u32,
    // KV heads (encode) or query heads (tq_rotate_q).
    n_heads: u32,
    head_dim: u32,
    // Elements per token row in `src` (kv_dim for encode, q_dim for rotate).
    src_stride: u32,
    // Cache timestep the first row of this batch writes (the chunk's start_pos).
    // Unused by tq_rotate_q.
    dst_pos: u32,
    // Cache capacity in timesteps — the per-head stride of every cache region.
    max_seq_len: u32,
    // Offset of this layer's polar signs in `signs`; JL signs follow at
    // `sign_off + head_dim`.
    sign_off: u32,
    // Row capacity of the rotated-query scratch buffer — the per-region stride
    // for tq_rotate_q. Unused by the encode kernels.
    q_cap: u32,
    // Lloyd-Max centroids (ascending) and the 3 decision boundaries between
    // them. Unused by tq_rotate_q.
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_words: array<u32>;
@group(0) @binding(2) var<storage, read> signs: array<f32>;
@group(0) @binding(3) var<storage, read> params: TqParams;

const MAX_HEAD_DIM: u32 = 128u;
const WG: u32 = 128u;
// Below this a vector is treated as zero (matches the CPU's `1e-12` guard).
const EPS: f32 = 1e-12;

var<workgroup> rot: array<f32, MAX_HEAD_DIM>;
var<workgroup> idxs: array<u32, MAX_HEAD_DIM>;
var<workgroup> red: array<f32, WG>;

// ── Shared helpers ─────────────────────────────────────────────────────────

// Nearest of the 4 Lloyd-Max centroids as a 2-bit index. Mirrors
// `turboquant::quantize_scalar`'s boundary comparisons exactly.
fn quantize_scalar(v: f32) -> u32 {
    if v < params.b1 {
        if v < params.b0 { return 0u; }
        return 1u;
    }
    if v < params.b2 { return 2u; }
    return 3u;
}

fn centroid_of(idx: u32) -> f32 {
    var c: array<f32, 4> = array<f32, 4>(params.c0, params.c1, params.c2, params.c3);
    return c[idx];
}

// ── tq_encode_keys ─────────────────────────────────────────────────────────

@compute @workgroup_size(WG, 1, 1)
fn tq_encode_keys(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let head_dim = params.head_dim;
    let n_kv_heads = params.n_heads;
    let idx = get_wid(wid);
    if idx >= params.n_tokens * n_kv_heads {
        return;
    }
    let j = idx / n_kv_heads;         // row within this batch
    let h = idx % n_kv_heads;         // KV head
    let t = params.dst_pos + j;       // absolute cache timestep

    let polar_words = head_dim / 16u;
    let jl_words = head_dim / 32u;
    let vecs = n_kv_heads * params.max_seq_len;
    let jl_off = vecs * polar_words;
    let norm_off = jl_off + vecs * jl_words;
    let slot = h * params.max_seq_len + t;
    let src_base = j * params.src_stride + h * head_dim;
    let inv_sqrt_d = 1.0 / sqrt(f32(head_dim));

    // ── 1. norm = ||k_head|| ──
    // Every lane writes its `red` slot: workgroup memory is not zero-initialized
    // (pipelines are built with `zero_initialize_workgroup_memory: false`).
    var x = 0.0;
    if tid < head_dim {
        x = src[src_base + tid];
    }
    red[tid] = x * x;
    workgroupBarrier();
    if tid < 64u { red[tid] += red[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { red[tid] += red[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { red[tid] += red[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { red[tid] += red[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { red[tid] += red[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { red[tid] += red[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { red[tid] += red[tid + 1u]; }
    workgroupBarrier();
    let norm = sqrt(red[0]);
    let is_zero = norm < EPS;
    // Reciprocal-multiply, not divide: the CPU reference computes
    // `k_head[i] * (1.0 / norm)`, and matching it removes a per-element 1-ulp
    // divergence from the packed indices. The `select` avoids 1/~0 for a zero
    // vector; its packed words are forced to 0 at the write sites anyway.
    let inv_norm = 1.0 / select(norm, 1.0, is_zero);

    // ── 2. normalize + PolarQuant RHT ──
    if tid < head_dim {
        rot[tid] = x * inv_norm * signs[params.sign_off + tid];
    }
    var stride = 1u;
    while stride < head_dim {
        workgroupBarrier();
        if tid < head_dim / 2u {
            let i = (tid / stride) * 2u * stride + (tid % stride);
            let a = rot[i];
            let b = rot[i + stride];
            rot[i] = a + b;
            rot[i + stride] = a - b;
        }
        stride = stride * 2u;
    }
    workgroupBarrier();
    if tid < head_dim {
        rot[tid] = rot[tid] * inv_sqrt_d;
    }
    workgroupBarrier();

    // ── 3. quantize, stash the residual in `rot`, reduce its squared norm ──
    var r = 0.0;
    if tid < head_dim {
        let qi = quantize_scalar(rot[tid]);
        idxs[tid] = qi;
        r = rot[tid] - centroid_of(qi);
    }
    red[tid] = r * r;
    workgroupBarrier();
    // Safe to overwrite `rot` now: every lane has read its own slot above, and
    // the residual is only read again after the reduction's barriers.
    if tid < head_dim {
        rot[tid] = r;
    }
    if tid < 64u { red[tid] += red[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { red[tid] += red[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { red[tid] += red[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { red[tid] += red[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { red[tid] += red[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { red[tid] += red[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { red[tid] += red[tid + 1u]; }
    workgroupBarrier();
    let residual_norm = sqrt(red[0]);
    let no_residual = residual_norm < EPS;
    // Reciprocal-multiply for the same reason as `inv_norm` above.
    let inv_rnorm = 1.0 / select(residual_norm, 1.0, no_residual);

    // Pack the 2-bit indices: one thread per output word, 16 elements each.
    if tid < polar_words {
        var w = 0u;
        for (var k = 0u; k < 16u; k += 1u) {
            w |= idxs[tid * 16u + k] << (2u * k);
        }
        out_words[slot * polar_words + tid] = select(w, 0u, is_zero);
    }

    // ── 4. QJL: normalize the residual, second RHT, pack sign bits ──
    if tid < head_dim {
        rot[tid] = rot[tid] * inv_rnorm * signs[params.sign_off + head_dim + tid];
    }
    var s = 1u;
    while s < head_dim {
        workgroupBarrier();
        if tid < head_dim / 2u {
            let i = (tid / s) * 2u * s + (tid % s);
            let a = rot[i];
            let b = rot[i + s];
            rot[i] = a + b;
            rot[i + s] = a - b;
        }
        s = s * 2u;
    }
    workgroupBarrier();
    // The 1/sqrt(head_dim) normalization is deliberately skipped here: it is a
    // positive scalar, and only the sign of each component survives into the
    // packed bits.
    if tid < jl_words {
        var w = 0u;
        for (var k = 0u; k < 32u; k += 1u) {
            if rot[tid * 32u + k] >= 0.0 {
                w |= 1u << k;
            }
        }
        out_words[jl_off + slot * jl_words + tid] = select(w, 0u, is_zero || no_residual);
    }

    if tid == 0u {
        let packed = pack2x16float(vec2<f32>(norm, residual_norm));
        out_words[norm_off + slot] = select(packed, 0u, is_zero);
    }
}

// ── tq_encode_values ───────────────────────────────────────────────────────
//
// PolarQuant only. Shares the key kernel's rotation and packing; there is no
// residual pass and no JL region.

@compute @workgroup_size(WG, 1, 1)
fn tq_encode_values(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let head_dim = params.head_dim;
    let n_kv_heads = params.n_heads;
    let idx = get_wid(wid);
    if idx >= params.n_tokens * n_kv_heads {
        return;
    }
    let j = idx / n_kv_heads;
    let h = idx % n_kv_heads;
    let t = params.dst_pos + j;

    let polar_words = head_dim / 16u;
    let norm_off = n_kv_heads * params.max_seq_len * polar_words;
    let slot = h * params.max_seq_len + t;
    let src_base = j * params.src_stride + h * head_dim;
    let inv_sqrt_d = 1.0 / sqrt(f32(head_dim));

    var x = 0.0;
    if tid < head_dim {
        x = src[src_base + tid];
    }
    red[tid] = x * x;
    workgroupBarrier();
    if tid < 64u { red[tid] += red[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { red[tid] += red[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { red[tid] += red[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { red[tid] += red[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { red[tid] += red[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { red[tid] += red[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { red[tid] += red[tid + 1u]; }
    workgroupBarrier();
    let norm = sqrt(red[0]);
    let is_zero = norm < EPS;
    let inv_norm = 1.0 / select(norm, 1.0, is_zero);

    if tid < head_dim {
        rot[tid] = x * inv_norm * signs[params.sign_off + tid];
    }
    var stride = 1u;
    while stride < head_dim {
        workgroupBarrier();
        if tid < head_dim / 2u {
            let i = (tid / stride) * 2u * stride + (tid % stride);
            let a = rot[i];
            let b = rot[i + stride];
            rot[i] = a + b;
            rot[i + stride] = a - b;
        }
        stride = stride * 2u;
    }
    workgroupBarrier();
    if tid < head_dim {
        idxs[tid] = quantize_scalar(rot[tid] * inv_sqrt_d);
    }
    workgroupBarrier();

    if tid < polar_words {
        var w = 0u;
        for (var k = 0u; k < 16u; k += 1u) {
            w |= idxs[tid * 16u + k] << (2u * k);
        }
        out_words[slot * polar_words + tid] = select(w, 0u, is_zero);
    }
    if tid == 0u {
        let packed = pack2x16float(vec2<f32>(norm, 0.0));
        out_words[norm_off + slot] = select(packed, 0u, is_zero);
    }
}

// ── tq_rotate_q ────────────────────────────────────────────────────────────
//
// Mirrors `turboquant::rotate_queries`: q_rot = RHT_polar(q), q_jl =
// RHT_jl(q_rot) (the residual lives in rotated space, so JL is applied to the
// already-rotated query), plus the per-head sum of q_jl the attention kernel
// needs to turn a positive-bit sum into a signed one.
//
// Unlike the encode kernels the query is NOT normalized — the estimator is
// linear in q, so the query's magnitude carries straight through to the score.
//
// Output regions in `out_words` (f32 values written via `bitcast`):
//   q_rot: (j * n_heads + h) * head_dim + d
//   q_jl:  region     + same
//   sums:  2 * region + j * n_heads + h
// where `region = q_cap * n_heads * head_dim`.

@compute @workgroup_size(WG, 1, 1)
fn tq_rotate_q(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let head_dim = params.head_dim;
    let n_heads = params.n_heads;
    let idx = get_wid(wid);
    if idx >= params.n_tokens * n_heads {
        return;
    }
    let j = idx / n_heads;
    let h = idx % n_heads;

    let region = params.q_cap * n_heads * head_dim;
    let out_base = (j * n_heads + h) * head_dim;
    let inv_sqrt_d = 1.0 / sqrt(f32(head_dim));

    if tid < head_dim {
        rot[tid] = src[j * params.src_stride + h * head_dim + tid]
            * signs[params.sign_off + tid];
    }
    var stride = 1u;
    while stride < head_dim {
        workgroupBarrier();
        if tid < head_dim / 2u {
            let i = (tid / stride) * 2u * stride + (tid % stride);
            let a = rot[i];
            let b = rot[i + stride];
            rot[i] = a + b;
            rot[i + stride] = a - b;
        }
        stride = stride * 2u;
    }
    workgroupBarrier();
    var q_rot_d = 0.0;
    if tid < head_dim {
        q_rot_d = rot[tid] * inv_sqrt_d;
        out_words[out_base + tid] = bitcast<u32>(q_rot_d);
    }
    workgroupBarrier();

    // Second RHT, applied to the rotated query.
    if tid < head_dim {
        rot[tid] = q_rot_d * signs[params.sign_off + head_dim + tid];
    }
    var s = 1u;
    while s < head_dim {
        workgroupBarrier();
        if tid < head_dim / 2u {
            let i = (tid / s) * 2u * s + (tid % s);
            let a = rot[i];
            let b = rot[i + s];
            rot[i] = a + b;
            rot[i + s] = a - b;
        }
        s = s * 2u;
    }
    workgroupBarrier();
    var q_jl_d = 0.0;
    if tid < head_dim {
        q_jl_d = rot[tid] * inv_sqrt_d;
        out_words[region + out_base + tid] = bitcast<u32>(q_jl_d);
    }

    // sum(q_jl) for this head.
    red[tid] = q_jl_d;
    workgroupBarrier();
    if tid < 64u { red[tid] += red[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { red[tid] += red[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { red[tid] += red[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { red[tid] += red[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { red[tid] += red[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { red[tid] += red[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { red[tid] += red[tid + 1u]; }
    workgroupBarrier();
    if tid == 0u {
        out_words[2u * region + j * n_heads + h] = bitcast<u32>(red[0]);
    }
}

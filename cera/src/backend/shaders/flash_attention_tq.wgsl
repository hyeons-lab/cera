// FlashAttention over a TurboQuant-compressed KV cache — GPU port of
// `turboquant::attn_scores_turboquant_gqa` + `attn_values_turboquant_gqa`.
//
// Structurally this is `attention_prefill.wgsl` with the two f32 KV reads
// replaced by the compressed estimators, so one kernel serves both paths:
// decode dispatches a single query row (`n_queries = 1`, `start_pos = pos`),
// chunked prefill dispatches the whole chunk. Online softmax (Dao 2022), one
// workgroup per (head, query), TILE-sized score tiles in workgroup memory, no
// materialized scores slab.
//
// ── Scores ─────────────────────────────────────────────────────────────────
// Keys are never reconstructed. With the query pre-rotated into the same basis
// by `tq_rotate_q`, the PolarQuant term is a direct dot against the 4 Lloyd-Max
// centroids in rotated space, and the QJL sign bits contribute an unbiased
// inner-product correction (arXiv:2504.19874 §3.2):
//
//   polar_dot  = norm * Σ_d q_rot[d] * centroid[idx(d)]
//   signed_sum = 2 * Σ_{d : jl_bit(d)=1} q_jl[d]  -  Σ_d q_jl[d]
//   correction = norm * residual_norm * sqrt(pi/2)/head_dim * signed_sum
//   score      = (polar_dot + correction) * scale
//
// `residual_norm` is stored in unit-normalized key space, so the correction is
// rescaled by `norm` to match `polar_dot` — same as the CPU scalar path.
//
// ── Values ─────────────────────────────────────────────────────────────────
// The RHT is linear, so the accumulator stays in *rotated* space for the whole
// tiled pass and the inverse rotation is applied exactly once, in the epilogue —
// not once per timestep. Online-softmax rescaling commutes with it (both are
// linear), which is what makes the flash formulation work here at all.
//
// ── Constraints (asserted host-side in encode_attention_tq) ────────────────
//   - head_dim <= 128 (bounds q_rot_shared / q_jl_shared / acc), a power of two
//     (Walsh-Hadamard), and a multiple of 32 (whole JL sign words).
//   - caller MUST pass `max_seq >= start_pos + n_queries`; the compressed cache
//     must hold valid entries for positions `[0, start_pos + n_queries)`. As a
//     defensive belt the shader clamps `seq_len = min(pos_q + 1, max_seq)`, so an
//     under-sized `max_seq` truncates the attention window rather than reading
//     out of bounds.
//   - keys AND values must both be compressed (the GPU path has no mixed mode).
// GQA: kv_head = head / (n_heads / n_kv_heads).
//
// All barriers sit at entry-point scope and the tree reductions are inlined —
// naga's SPIR-V path miscompiles `workgroupBarrier()` reached through a function
// call inside a loop. Workgroup memory is not zero-initialized, so every slot is
// written before it is read.
//
// Cache layout is documented in `turboquant.wgsl` (regions, LSB-first packing,
// f16 norms via pack2x16float).
//
// Bind group 0:
//   @binding(0) qrot:      array<f32>  rotated queries: [q_rot | q_jl | sums]
//   @binding(1) k_cache:   array<u32>  [polar | jl | norms]
//   @binding(2) v_cache:   array<u32>  [polar | norms]
//   @binding(3) out_batch: array<f32>  n_queries × out_stride floats (rw)
//   @binding(4) params:    array<u32, 16>
//        ( n_heads, n_kv_heads, head_dim, max_seq, start_pos, scale_bits,
//          q_cap, out_stride, qjl_scale_bits, sign_off,
//          c0_bits, c1_bits, c2_bits, c3_bits, q_base, cache_cap )
//
// `max_seq` is the causal clamp (`start_pos + n_queries`); `cache_cap` is the
// cache's allocated timestep capacity, which is the per-head stride of every
// compressed region. The two are NOT interchangeable — the f32 kernels get away
// with one value because their KV rows are addressed by `kv_dim` alone.
//   @binding(5) signs:     array<f32>  all layers' [polar | jl] sign flips
//
// Dispatch: (n_heads, n_queries, 1) workgroups of 256 threads.

@group(0) @binding(0) var<storage, read> qrot: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<u32>;
@group(0) @binding(2) var<storage, read> v_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_batch: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32, 16>;
@group(0) @binding(5) var<storage, read> signs: array<f32>;

const TILE: u32 = 256u;
const MAX_HEAD_DIM: u32 = 128u;
const NEG_INF: f32 = -3.402823e+38;

var<workgroup> q_rot_shared: array<f32, MAX_HEAD_DIM>;
var<workgroup> q_jl_shared: array<f32, MAX_HEAD_DIM>;
var<workgroup> acc: array<f32, MAX_HEAD_DIM>;   // rotated-space output accumulator
var<workgroup> tile_scores: array<f32, TILE>;
// This tile's per-timestep value norms, staged by the thread that scored the
// timestep. Without it every one of the `head_dim` accumulator threads would
// re-read the same norm word for every timestep in the tile.
var<workgroup> tile_vnorm: array<f32, TILE>;
var<workgroup> red: array<f32, TILE>;           // reduction scratch
// Running online-softmax state, broadcast to all threads via workgroup memory.
// [0]=running max, [1]=running sum, [2]=this tile's new max, [3]=correction.
var<workgroup> st: array<f32, 4>;

@compute @workgroup_size(256, 1, 1)
fn flash_attention_tq(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.x;
    let q_idx = wid.y;
    let tid = lid.x;

    let n_heads = params[0];
    let n_kv_heads = params[1];
    let head_dim = params[2];
    let max_seq = params[3];
    let start_pos = params[4];
    let scale = bitcast<f32>(params[5]);
    let q_cap = params[6];
    let out_stride = params[7];
    let qjl_scale = bitcast<f32>(params[8]);
    let sign_off = params[9];
    var centroids: array<f32, 4> = array<f32, 4>(
        bitcast<f32>(params[10]),
        bitcast<f32>(params[11]),
        bitcast<f32>(params[12]),
        bitcast<f32>(params[13]),
    );
    let q_base = params[14];
    let cache_cap = params[15];

    let q_global = q_base + q_idx;
    // Per-query causal window: attend over [0..pos_q]. Clamped against max_seq so
    // inconsistent params can only truncate the window, never read OOB.
    let pos_q = start_pos + q_global;
    let seq_len = min(pos_q + 1u, max_seq);

    let group_size = n_heads / n_kv_heads;
    let kv_head = head / group_size;
    let out_offset = q_global * out_stride + head * head_dim;

    // Rotated-query regions (see `tq_rotate_q`).
    let q_region = q_cap * n_heads * head_dim;
    let q_offset = (q_global * n_heads + head) * head_dim;
    let q_jl_sum = qrot[2u * q_region + q_global * n_heads + head];

    // Compressed cache regions for this layer.
    let polar_words = head_dim / 16u;
    let jl_words = head_dim / 32u;
    let vecs = n_kv_heads * cache_cap;
    let k_jl_off = vecs * polar_words;
    let k_norm_off = k_jl_off + vecs * jl_words;
    let v_norm_off = vecs * polar_words;
    let kv_slot_base = kv_head * cache_cap;

    // seq_len == 0 would divide by st[1] == 0 → NaN. Write zeros and bail.
    // `seq_len` depends only on params and the (uniform) workgroup id.
    if seq_len == 0u {
        if tid < head_dim {
            out_batch[out_offset + tid] = 0.0;
        }
        return;
    }

    if tid < head_dim {
        q_rot_shared[tid] = qrot[q_offset + tid];
        q_jl_shared[tid] = qrot[q_region + q_offset + tid];
        acc[tid] = 0.0;
    }
    if tid == 0u {
        st[0] = NEG_INF; // running max
        st[1] = 0.0;     // running sum
    }
    workgroupBarrier();

    var base = 0u;
    while base < seq_len {
        // ── score for timestep t = base + tid (one per thread) ──
        let t = base + tid;
        var score = NEG_INF;
        if t < seq_len {
            let slot = kv_slot_base + t;
            let norms = unpack2x16float(k_cache[k_norm_off + slot]);
            let norm = norms.x;
            let residual_norm = norms.y;

            // PolarQuant: dot the rotated query against the centroid the 2-bit
            // index selects, 16 elements per packed word.
            var polar_dot = 0.0;
            let polar_base = slot * polar_words;
            for (var w = 0u; w < polar_words; w += 1u) {
                let word = k_cache[polar_base + w];
                let d0 = w * 16u;
                for (var k = 0u; k < 16u; k += 1u) {
                    polar_dot += q_rot_shared[d0 + k] * centroids[(word >> (2u * k)) & 3u];
                }
            }

            // QJL: sum the JL-projected query over the set sign bits, then turn
            // that positive-only sum into the signed one via the precomputed
            // total (2 * pos_sum - total).
            var pos_sum = 0.0;
            let jl_base = k_jl_off + slot * jl_words;
            for (var w = 0u; w < jl_words; w += 1u) {
                let word = k_cache[jl_base + w];
                let d0 = w * 32u;
                for (var k = 0u; k < 32u; k += 1u) {
                    pos_sum += q_jl_shared[d0 + k] * f32((word >> k) & 1u);
                }
            }
            let signed_sum = 2.0 * pos_sum - q_jl_sum;

            let correction = norm * residual_norm * qjl_scale * signed_sum;
            score = (polar_dot * norm + correction) * scale;

            // Stage this timestep's value norm for the accumulation phase below.
            // Only slots with `tt < seq_len` are read there, and those are exactly
            // the slots written here.
            tile_vnorm[tid] = unpack2x16float(v_cache[v_norm_off + slot]).x;
        }
        tile_scores[tid] = score;

        // ── tile max (inlined tree reduction over `red`) ──
        red[tid] = score;
        workgroupBarrier();
        if tid < 128u { red[tid] = max(red[tid], red[tid + 128u]); }
        workgroupBarrier();
        if tid < 64u { red[tid] = max(red[tid], red[tid + 64u]); }
        workgroupBarrier();
        if tid < 32u { red[tid] = max(red[tid], red[tid + 32u]); }
        workgroupBarrier();
        if tid < 16u { red[tid] = max(red[tid], red[tid + 16u]); }
        workgroupBarrier();
        if tid < 8u { red[tid] = max(red[tid], red[tid + 8u]); }
        workgroupBarrier();
        if tid < 4u { red[tid] = max(red[tid], red[tid + 4u]); }
        workgroupBarrier();
        if tid < 2u { red[tid] = max(red[tid], red[tid + 2u]); }
        workgroupBarrier();
        if tid < 1u { red[tid] = max(red[tid], red[tid + 1u]); }
        workgroupBarrier();
        let tmax = red[0];

        // new running max + correction factor (published by thread 0)
        if tid == 0u {
            let nm = max(st[0], tmax);
            st[2] = nm;
            st[3] = exp(st[0] - nm); // first tile: exp(-inf) = 0
        }
        workgroupBarrier();
        let nm = st[2];
        let corr = st[3];

        // p = exp(score - nm); reuse tile_scores to hold the exponentials.
        var p = 0.0;
        if t < seq_len {
            p = exp(tile_scores[tid] - nm);
        }
        tile_scores[tid] = p;

        // ── tile sum (inlined tree reduction over `red`) ──
        red[tid] = p;
        workgroupBarrier();
        if tid < 128u { red[tid] += red[tid + 128u]; }
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
        let tsum = red[0];

        // Rescale the accumulator and add this tile's values — still in rotated
        // space. Thread `tid` owns rotated dim `tid`, so it needs one 2-bit field
        // out of each timestep's packed value vector.
        if tid < head_dim {
            var a = acc[tid] * corr;
            let vw = tid / 16u;              // word holding this dim
            let vshift = (tid % 16u) * 2u;   // bit offset within that word
            for (var jj = 0u; jj < TILE; jj += 1u) {
                let tt = base + jj;
                if tt < seq_len {
                    let word = v_cache[(kv_slot_base + tt) * polar_words + vw];
                    a += tile_scores[jj] * tile_vnorm[jj] * centroids[(word >> vshift) & 3u];
                }
            }
            acc[tid] = a;
        }
        if tid == 0u {
            st[1] = st[1] * corr + tsum;
            st[0] = nm;
        }
        // Barrier before the next tile reuses tile_scores/red and reads acc/st.
        workgroupBarrier();
        base += TILE;
    }

    // ── Epilogue: normalize, then one inverse RHT back to the original basis ──
    // `rht_inverse` is: scale by 1/sqrt(head_dim), Walsh-Hadamard (self-inverse),
    // undo the sign flip. Folding the softmax denominator in here is free.
    let inv_sqrt_d = 1.0 / sqrt(f32(head_dim));
    if tid < head_dim {
        acc[tid] = acc[tid] / st[1] * inv_sqrt_d;
    }
    var stride = 1u;
    while stride < head_dim {
        workgroupBarrier();
        if tid < head_dim / 2u {
            let i = (tid / stride) * 2u * stride + (tid % stride);
            let a = acc[i];
            let b = acc[i + stride];
            acc[i] = a + b;
            acc[i + stride] = a - b;
        }
        stride = stride * 2u;
    }
    workgroupBarrier();
    if tid < head_dim {
        out_batch[out_offset + tid] = acc[tid] * signs[sign_off + tid];
    }
}

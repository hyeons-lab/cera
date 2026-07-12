// FlashAttention (Dao 2022) for one query vector — the decode path.
//
// Online-softmax, single tiled pass over the KV cache with bounded workgroup
// memory (TILE scores, not seq_len). Replaces the classic `attention.wgsl`
// which materializes all scores into an external `scores_buf` scratch and runs
// three passes; flash needs no scratch buffer (5 bindings vs 6) and reads each
// K/V row once.
//
// One workgroup per head, 256 threads, TILE=256 (one KV timestep per thread per
// tile). Results match the classic kernel up to floating-point roundoff (online
// vs batched softmax accumulation order).
//
// The tree reductions are inlined (not helper functions) and all barriers sit at
// kernel scope, mirroring attention.wgsl / the gemv kernels — naga's SPIR-V path
// (lavapipe/Vulkan) miscompiles `workgroupBarrier()` reached through a function
// call inside a loop, so this file keeps every barrier in the entry point.
//
// Constraints (asserted host-side in encode_attention):
//   - head_dim <= 128 (bounds `q_shared` and `acc`).
//   - decode: the single query attends all cached KV, so no causal mask.
// GQA: kv_head = head / (n_heads / n_kv_heads).
//
// Bind group 0:
//   @binding(0) q: array<f32>        (all heads concatenated, read)
//   @binding(1) k_cache: array<f32>  (seq_len × kv_dim, read)
//   @binding(2) v_cache: array<f32>  (seq_len × kv_dim, read)
//   @binding(3) out: array<f32>      (all heads concatenated, read-write)
//   @binding(4) params: array<u32,8> (n_heads, n_kv_heads, head_dim, kv_dim,
//                                      seq_len, scale_bits, _, _)
// Dispatch: (n_heads, 1, 1).

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32, 8>;

const TILE: u32 = 256u;
const MAX_HEAD_DIM: u32 = 128u;
const NEG_INF: f32 = -3.402823e+38;

var<workgroup> q_shared: array<f32, MAX_HEAD_DIM>;
var<workgroup> acc: array<f32, MAX_HEAD_DIM>;   // per-dim output accumulator
var<workgroup> tile_scores: array<f32, TILE>;
var<workgroup> red: array<f32, TILE>;           // reduction scratch
// Running online-softmax state, broadcast to all threads via workgroup memory.
// [0]=running max, [1]=running sum, [2]=this tile's new max, [3]=correction.
var<workgroup> st: array<f32, 4>;

@compute @workgroup_size(256, 1, 1)
fn flash_attention(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.x;
    let tid = lid.x;
    let n_heads = params[0];
    let n_kv_heads = params[1];
    let head_dim = params[2];
    let kv_dim = params[3];
    let seq_len = params[4];
    let scale = bitcast<f32>(params[5]);

    let group_size = n_heads / n_kv_heads;
    let kv_head = head / group_size;
    let kv_h_offset = kv_head * head_dim;
    let q_offset = head * head_dim;

    // seq_len == 0 would divide by st[1] == 0 → NaN. Write zeros and bail.
    if seq_len == 0u {
        if tid < head_dim {
            out[q_offset + tid] = 0.0;
        }
        return;
    }

    // Load Q into workgroup memory and zero the output accumulator.
    if tid < head_dim {
        q_shared[tid] = q[q_offset + tid];
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
            var dot = 0.0;
            let k_base = t * kv_dim + kv_h_offset;
            for (var d = 0u; d < head_dim; d += 1u) {
                dot += q_shared[d] * k_cache[k_base + d];
            }
            score = dot * scale;
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

        // rescale the accumulator by the correction and add this tile's V.
        if tid < head_dim {
            var a = acc[tid] * corr;
            let vd = kv_h_offset + tid;
            for (var jj = 0u; jj < TILE; jj += 1u) {
                let tt = base + jj;
                if tt < seq_len {
                    a += tile_scores[jj] * v_cache[tt * kv_dim + vd];
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

    if tid < head_dim {
        out[q_offset + tid] = acc[tid] / st[1];
    }
}

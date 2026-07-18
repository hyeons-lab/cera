// Batched FlashAttention prefill: N queries × n_heads in one dispatch.
//
// Online-softmax (Dao 2022), single tiled pass over the KV cache with bounded
// workgroup memory (TILE scores, not seq_len) and NO external scores scratch
// buffer — 5 bindings, each K/V row read once per (query, head). This is the
// decode `flash_attention.wgsl` kernel extended along the query dimension: one
// workgroup per (head, query), a per-query causal window, and batched Q/out
// strides. Matches a plain batched-softmax reference up to floating-point
// roundoff (online vs batched accumulation order); the wgpu-side test asserts it
// against a CPU ground truth.
//
// Because the scores are never materialized (only a TILE-sized tile lives in
// workgroup memory), the storage bindings do not scale with seq_len — this
// removes the `[n_queries, n_heads, max_seq]` scores slab that previously forced
// host-side query-tiling to stay under the adapter storage-binding limit, and it
// gives K/V reuse across the tiled pass. Contexts long enough that the *KV*
// binding itself (`max_seq × kv_dim`) overflows the limit remain future
// paged-KV work, guarded host-side.
//
// The tree reductions are inlined (not helper functions) and all barriers sit at
// kernel scope — naga's SPIR-V path (lavapipe/Vulkan) miscompiles
// `workgroupBarrier()` reached through a function call inside a loop, so every
// barrier stays in the entry point.
//
// Constraints (asserted host-side in encode_attention_prefill):
//   - head_dim <= 128 (bounds `q_shared` and `acc`).
//   - caller MUST pass `max_seq >= start_pos + n_queries`; K/V must hold valid
//     entries for positions `[0, start_pos + n_queries)`. As a defensive belt the
//     shader clamps `seq_len = min(pos_q + 1, max_seq)` — an under-sized `max_seq`
//     yields truncated (incorrect) attention rather than an OOB read.
// GQA: kv_head = head / (n_heads / n_kv_heads).
//
// Bind group 0:
//   @binding(0) q_batch:   array<f32>    n_queries × q_stride floats
//   @binding(1) k_cache:   array<f32>    seq_len × kv_dim floats
//   @binding(2) v_cache:   array<f32>    seq_len × kv_dim floats
//   @binding(3) out_batch: array<f32>    n_queries × out_stride floats (rw)
//   @binding(4) params:    array<u32, 12>
//        ( n_heads, n_kv_heads, head_dim, kv_dim, max_seq, scale_bits,
//          start_pos, n_queries, q_stride, out_stride, q_base, _pad1 )
//
// `q_base` is the index, within `q_batch` / `out_batch`, of the first query in
// this dispatch, so a caller may still split the query batch across dispatches;
// `q_global = q_base + q_idx` addresses `q_batch` / `out_batch` and sets the
// causal position. `q_base = 0` is the single-dispatch case (the default now that
// the scores slab is gone).
//
// Dispatch: (n_heads, n_queries, 1) workgroups of 256 threads.

@group(0) @binding(0) var<storage, read> q_batch: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_batch: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32, 12>;

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
fn attention_prefill(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.x;
    let q_idx = wid.y;
    let tid = lid.x;

    let n_heads = params[0];
    let n_kv_heads = params[1];
    let head_dim = params[2];
    let kv_dim = params[3];
    let max_seq = params[4];
    let scale = bitcast<f32>(params[5]);
    let start_pos = params[6];
    // params[7] (n_queries) is implicit in dispatch.
    let q_stride = params[8];
    let out_stride = params[9];
    let q_base = params[10];

    // Global query index into q_batch / out_batch (q_base + local q_idx).
    let q_global = q_base + q_idx;

    // Per-query causal window: attend over [0..pos_q]. Clamp against max_seq so a
    // caller passing inconsistent params can only cause silent window truncation,
    // never an OOB read of k_cache / v_cache.
    let pos_q = start_pos + q_global;
    let seq_len = min(pos_q + 1u, max_seq);

    let group_size = n_heads / n_kv_heads;
    let kv_head = head / group_size;
    let kv_h_offset = kv_head * head_dim;
    let q_offset = q_global * q_stride + head * head_dim;
    let out_offset = q_global * out_stride + head * head_dim;

    // seq_len == 0 would divide by st[1] == 0 → NaN. Write zeros and bail.
    if seq_len == 0u {
        if tid < head_dim {
            out_batch[out_offset + tid] = 0.0;
        }
        return;
    }

    // Load Q into workgroup memory and zero the output accumulator.
    if tid < head_dim {
        q_shared[tid] = q_batch[q_offset + tid];
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
        out_batch[out_offset + tid] = acc[tid] / st[1];
    }
}
